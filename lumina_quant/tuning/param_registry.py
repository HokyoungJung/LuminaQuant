"""Strategy hyper-parameter registry utilities.

This module provides:
- ``HyperParam``: typed parameter metadata + runtime coercion.
- ``ParamRegistry``: strategy-scoped schema registry and search-space builders.

The canonical naming scheme for cross-strategy parameter identifiers is:
``<strategy_slug>.<param_name>``.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

_SNAKE_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_CAMEL_TOKEN_PATTERN = re.compile(r"(?<!^)(?=[A-Z])")

_TRUE_TOKENS = {"1", "true", "yes", "on", "y", "t"}
_FALSE_TOKENS = {"0", "false", "no", "off", "n", "f"}


def strategy_slug(strategy_name: str) -> str:
    token = str(strategy_name or "").strip()
    if token.lower().endswith("strategy"):
        token = token[: -len("strategy")]
    snake = _CAMEL_TOKEN_PATTERN.sub("_", token).strip("_").lower()
    return snake or "strategy"


def canonical_param_name(strategy_name: str, param_name: str) -> str:
    return f"{strategy_slug(strategy_name)}.{str(param_name).strip()}"


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
    return bool(default)


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _coerce_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_int_tuple(
    value: Any,
    default: Sequence[int],
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> tuple[int, ...]:
    candidates: list[Any]
    if value is None:
        candidates = list(default)
    elif isinstance(value, str):
        candidates = [part.strip() for part in value.split(",")]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, str)):
        candidates = list(value)
    else:
        candidates = [value]

    out: list[int] = []
    for item in candidates:
        try:
            parsed = int(item)
        except Exception:
            continue
        if min_value is not None and parsed < min_value:
            continue
        if max_value is not None and parsed > max_value:
            continue
        out.append(parsed)

    deduped: list[int] = []
    seen: set[int] = set()
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)

    return tuple(deduped) if deduped else tuple(int(x) for x in default)


@dataclass(frozen=True, slots=True)
class HyperParam:
    """Single strategy hyper-parameter definition."""

    name: str
    kind: str
    default: Any
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    choices: tuple[Any, ...] | None = None
    tunable: bool = True
    optuna: Mapping[str, Any] | None = None
    grid: Sequence[Any] | None = None
    description: str = ""
    parser: Callable[[Any, Any], Any] | None = None

    @classmethod
    def integer(
        cls,
        name: str,
        default: int,
        *,
        low: int | None = None,
        high: int | None = None,
        step: int | None = None,
        tunable: bool = True,
        optuna: Mapping[str, Any] | None = None,
        grid: Sequence[Any] | None = None,
        description: str = "",
    ) -> HyperParam:
        return cls(
            name=name,
            kind="int",
            default=int(default),
            low=low,
            high=high,
            step=step,
            tunable=tunable,
            optuna=optuna,
            grid=grid,
            description=description,
        )

    @classmethod
    def floating(
        cls,
        name: str,
        default: float,
        *,
        low: float | None = None,
        high: float | None = None,
        step: float | None = None,
        tunable: bool = True,
        optuna: Mapping[str, Any] | None = None,
        grid: Sequence[Any] | None = None,
        description: str = "",
    ) -> HyperParam:
        return cls(
            name=name,
            kind="float",
            default=float(default),
            low=low,
            high=high,
            step=step,
            tunable=tunable,
            optuna=optuna,
            grid=grid,
            description=description,
        )

    @classmethod
    def boolean(
        cls,
        name: str,
        default: bool,
        *,
        tunable: bool = True,
        optuna: Mapping[str, Any] | None = None,
        grid: Sequence[Any] | None = None,
        description: str = "",
    ) -> HyperParam:
        return cls(
            name=name,
            kind="bool",
            default=bool(default),
            tunable=tunable,
            optuna=optuna,
            grid=grid,
            description=description,
        )

    @classmethod
    def categorical(
        cls,
        name: str,
        default: Any,
        *,
        choices: Sequence[Any],
        tunable: bool = True,
        optuna: Mapping[str, Any] | None = None,
        grid: Sequence[Any] | None = None,
        description: str = "",
    ) -> HyperParam:
        choice_tuple = tuple(choices)
        return cls(
            name=name,
            kind="categorical",
            default=copy.deepcopy(default),
            choices=choice_tuple,
            tunable=tunable,
            optuna=optuna,
            grid=grid,
            description=description,
        )

    @classmethod
    def string(
        cls,
        name: str,
        default: str = "",
        *,
        tunable: bool = False,
        description: str = "",
    ) -> HyperParam:
        return cls(
            name=name,
            kind="str",
            default=str(default),
            tunable=tunable,
            description=description,
        )

    @classmethod
    def int_tuple(
        cls,
        name: str,
        default: Sequence[int],
        *,
        min_value: int = 1,
        max_value: int | None = None,
        tunable: bool = False,
        description: str = "",
    ) -> HyperParam:
        default_tuple = tuple(int(item) for item in default)

        def _parser(value: Any, fallback: Any) -> tuple[int, ...]:
            return _coerce_int_tuple(
                value,
                tuple(int(x) for x in fallback),
                min_value=min_value,
                max_value=max_value,
            )

        return cls(
            name=name,
            kind="int_tuple",
            default=default_tuple,
            tunable=tunable,
            description=description,
            parser=_parser,
        )

    def with_name(self, name: str) -> HyperParam:
        return replace(self, name=str(name))

    def resolve(self, value: Any) -> Any:
        fallback = copy.deepcopy(self.default)
        raw = fallback if value is None else value
        if self.parser is not None:
            try:
                return self.parser(raw, fallback)
            except Exception:
                return copy.deepcopy(fallback)

        if self.kind == "int":
            out = _coerce_int(raw, int(fallback))
            if self.low is not None:
                out = max(int(self.low), out)
            if self.high is not None:
                out = min(int(self.high), out)
            return out

        if self.kind == "float":
            out = _coerce_float(raw, float(fallback))
            if self.low is not None:
                out = max(float(self.low), out)
            if self.high is not None:
                out = min(float(self.high), out)
            return out

        if self.kind == "bool":
            return _coerce_bool(raw, bool(fallback))

        if self.kind == "str":
            try:
                return str(raw)
            except Exception:
                return str(fallback)

        if self.kind == "categorical":
            choices = tuple(self.choices or ())
            if raw in choices:
                return raw
            return copy.deepcopy(fallback)

        if self.kind == "int_tuple":
            return copy.deepcopy(fallback)

        return copy.deepcopy(fallback)

    def to_optuna_spec(self) -> dict[str, Any] | None:
        if not self.tunable:
            return None
        if self.optuna is not None:
            return copy.deepcopy(dict(self.optuna))

        if self.kind == "int":
            if self.low is None or self.high is None:
                return None
            spec: dict[str, Any] = {"type": "int", "low": int(self.low), "high": int(self.high)}
            if self.step is not None:
                spec["step"] = int(self.step)
            return spec

        if self.kind == "float":
            if self.low is None or self.high is None:
                return None
            spec = {"type": "float", "low": float(self.low), "high": float(self.high)}
            if self.step is not None:
                spec["step"] = float(self.step)
            return spec

        if self.kind == "bool":
            return {"type": "categorical", "choices": [True, False]}

        if self.kind == "categorical":
            choices = list(self.choices or ())
            if not choices:
                return None
            return {"type": "categorical", "choices": choices}

        return None

    def to_grid_values(self) -> list[Any] | None:
        if not self.tunable:
            return None
        if self.grid is not None:
            return [copy.deepcopy(item) for item in self.grid]

        if self.kind == "bool":
            return [True, False]
        if self.kind == "categorical":
            choices = list(self.choices or ())
            return [copy.deepcopy(item) for item in choices] if choices else None
        if self.kind == "int" and self.low is not None and self.high is not None:
            low = int(self.low)
            high = int(self.high)
            mid = int(self.resolve(self.default))
            values = sorted({low, mid, high})
            return values
        if self.kind == "float" and self.low is not None and self.high is not None:
            low = float(self.low)
            high = float(self.high)
            mid = float(self.resolve(self.default))
            values = sorted({low, mid, high})
            return values
        return None


def resolve_params_from_schema(
    schema: Mapping[str, HyperParam],
    overrides: Mapping[str, Any] | None = None,
    *,
    keep_unknown: bool = True,
) -> dict[str, Any]:
    source = dict(overrides or {})
    resolved: dict[str, Any] = {}
    used_keys: set[str] = set()
    for key, param in schema.items():
        has_value = key in source
        raw = source.get(key)
        if has_value:
            used_keys.add(key)
        resolved[key] = param.resolve(raw if has_value else None)

    if keep_unknown:
        for key, value in source.items():
            if key in used_keys:
                continue
            if key in resolved:
                continue
            resolved[key] = copy.deepcopy(value)
    return resolved


@dataclass(slots=True)
class _StrategySchemaBundle:
    schema: dict[str, HyperParam]
    optuna_trials: int


class ParamRegistry:
    """Strategy parameter registry with schema + search-space helpers."""

    def __init__(self):
        self._bundles: dict[str, _StrategySchemaBundle] = {}
        self._runtime_values: dict[str, float] = {}

    # Runtime override key-value registry helpers (used by Alpha101 constants).
    def update(self, mapping: Mapping[str, Any]) -> None:
        for key, value in mapping.items():
            try:
                self._runtime_values[str(key)] = float(value)
            except Exception:
                continue

    def set(self, key: str, value: Any) -> None:
        try:
            self._runtime_values[str(key)] = float(value)
        except Exception:
            return

    def get(self, key: str, default: Any = 0.0) -> float:
        try:
            fallback = float(default)
        except Exception:
            fallback = 0.0
        return float(self._runtime_values.get(str(key), fallback))

    def clear_prefix(self, prefix: str = "") -> None:
        prefix_s = str(prefix)
        if not prefix_s:
            self._runtime_values.clear()
            return
        keys = [key for key in self._runtime_values if key.startswith(prefix_s)]
        for key in keys:
            self._runtime_values.pop(key, None)

    def snapshot(self, prefix: str = "") -> dict[str, float]:
        prefix_s = str(prefix)
        if not prefix_s:
            return dict(self._runtime_values)
        return {key: value for key, value in self._runtime_values.items() if key.startswith(prefix_s)}

    def register(
        self,
        strategy_name: str,
        schema: Mapping[str, HyperParam],
        *,
        optuna_trials: int = 20,
    ) -> None:
        name = str(strategy_name).strip()
        if not name:
            raise ValueError("strategy_name is required")
        normalized: dict[str, HyperParam] = {}
        for raw_key, raw_param in schema.items():
            key = str(raw_key).strip()
            if not _SNAKE_CASE_PATTERN.match(key):
                raise ValueError(f"Invalid parameter name '{key}' for strategy '{name}'")
            if not isinstance(raw_param, HyperParam):
                raise TypeError(
                    f"Schema '{name}.{key}' must be HyperParam, got {type(raw_param).__name__}"
                )
            normalized[key] = raw_param.with_name(key)
        self._bundles[name] = _StrategySchemaBundle(
            schema=normalized,
            optuna_trials=max(1, int(optuna_trials)),
        )

    def has_strategy(self, strategy_name: str) -> bool:
        return str(strategy_name).strip() in self._bundles

    def get_schema(self, strategy_name: str) -> dict[str, HyperParam]:
        bundle = self._bundles.get(str(strategy_name).strip())
        if bundle is None:
            return {}
        return dict(bundle.schema)

    def get_canonical_names(self, strategy_name: str) -> dict[str, str]:
        name = str(strategy_name).strip()
        bundle = self._bundles.get(name)
        if bundle is None:
            return {}
        return {
            param_name: canonical_param_name(name, param_name) for param_name in bundle.schema
        }

    def resolve_params(
        self,
        strategy_name: str,
        overrides: Mapping[str, Any] | None = None,
        *,
        keep_unknown: bool = True,
    ) -> dict[str, Any]:
        name = str(strategy_name).strip()
        bundle = self._bundles.get(name)
        if bundle is None:
            return dict(overrides or {})

        source = dict(overrides or {})
        return resolve_params_from_schema(bundle.schema, source, keep_unknown=keep_unknown)

    def default_params(self, strategy_name: str) -> dict[str, Any]:
        return self.resolve_params(strategy_name, {}, keep_unknown=False)

    def default_optuna_config(self, strategy_name: str) -> dict[str, Any]:
        name = str(strategy_name).strip()
        bundle = self._bundles.get(name)
        if bundle is None:
            return {}

        params: dict[str, Any] = {}
        for param_name, param in bundle.schema.items():
            spec = param.to_optuna_spec()
            if spec:
                params[param_name] = spec
        return {"n_trials": int(bundle.optuna_trials), "params": params}

    def default_grid_config(self, strategy_name: str) -> dict[str, Any]:
        name = str(strategy_name).strip()
        bundle = self._bundles.get(name)
        if bundle is None:
            return {}
        params: dict[str, list[Any]] = {}
        for param_name, param in bundle.schema.items():
            values = param.to_grid_values()
            if values:
                params[param_name] = values
        return {"params": params}

    def resolve_optuna_config(
        self,
        strategy_name: str,
        override: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        base = self.default_optuna_config(strategy_name)
        if not base:
            return {}
        if not isinstance(override, Mapping):
            return base

        out = copy.deepcopy(base)
        try:
            trials = int(override.get("n_trials", out["n_trials"]))
            out["n_trials"] = max(1, trials)
        except Exception:
            pass

        user_params = override.get("params")
        if not isinstance(user_params, Mapping):
            return out

        schema = self.get_schema(strategy_name)
        for key, value in user_params.items():
            if key not in schema or not isinstance(value, Mapping):
                continue
            existing = out["params"].get(key)
            if not isinstance(existing, dict):
                continue
            merged = dict(existing)
            merged.update(dict(value))
            out["params"][key] = merged
        return out

    def resolve_grid_config(
        self,
        strategy_name: str,
        override: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        base = self.default_grid_config(strategy_name)
        if not base:
            return {}
        if not isinstance(override, Mapping):
            return base

        out = copy.deepcopy(base)
        user_params = override.get("params")
        if not isinstance(user_params, Mapping):
            return out

        schema = self.get_schema(strategy_name)
        for key, values in user_params.items():
            if key not in schema:
                continue
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                continue
            cleaned = [schema[key].resolve(value) for value in values]
            if cleaned:
                out["params"][key] = cleaned
        return out
