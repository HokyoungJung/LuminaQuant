"""Public-safe strategy registry with optional private extension overlay."""

from __future__ import annotations

from copy import deepcopy
from importlib import import_module
from typing import Any

from lumina_quant.strategy import Strategy

from .sample_public_strategy import PublicSampleStrategy

StrategyClass = type[Strategy]

DEFAULT_STRATEGY_NAME = "PublicSampleStrategy"

_STRATEGY_MAP: dict[str, StrategyClass] = {
    "PublicSampleStrategy": PublicSampleStrategy,
    # Public build exposes sample strategy as aliases for private families.
    "RsiStrategy": PublicSampleStrategy,
    "MovingAverageCrossStrategy": PublicSampleStrategy,
    "RareEventScoreStrategy": PublicSampleStrategy,
}

_STRATEGY_TIER: dict[str, str] = dict.fromkeys(_STRATEGY_MAP, "live_default")

_PARAM_SCHEMAS: dict[str, dict[str, dict[str, Any]]] = {
    "PublicSampleStrategy": {
        "decision_cadence_seconds": {"type": "int", "min": 1, "max": 3600},
    },
    "RsiStrategy": {
        "rsi_period": {"type": "int", "min": 2, "max": 500},
        "oversold": {"type": "float", "min": 1.0, "max": 99.0},
        "overbought": {"type": "float", "min": 1.0, "max": 99.0},
        "allow_short": {"type": "bool"},
    },
    "MovingAverageCrossStrategy": {
        "short_window": {"type": "int", "min": 2, "max": 500},
        "long_window": {"type": "int", "min": 3, "max": 1000},
        "allow_short": {"type": "bool"},
    },
    "RareEventScoreStrategy": {
        "zscore_threshold": {"type": "float", "min": 0.1, "max": 20.0},
        "hold_bars": {"type": "int", "min": 1, "max": 10000},
        "allow_short": {"type": "bool"},
    },
}

_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "PublicSampleStrategy": {
        "decision_cadence_seconds": 20,
    },
    "RsiStrategy": {
        "rsi_period": 14,
        "oversold": 30.0,
        "overbought": 70.0,
        "allow_short": True,
    },
    "MovingAverageCrossStrategy": {
        "short_window": 20,
        "long_window": 50,
        "allow_short": True,
    },
    "RareEventScoreStrategy": {
        "zscore_threshold": 2.0,
        "hold_bars": 20,
        "allow_short": False,
    },
}

_CANONICAL_PARAM_NAMES: dict[str, dict[str, str]] = {
    "PublicSampleStrategy": {
        "decision_cadence_seconds": "public_sample.decision_cadence_seconds",
    },
    "RsiStrategy": {
        "rsi_period": "rsi.rsi_period",
        "oversold": "rsi.oversold",
        "overbought": "rsi.overbought",
        "allow_short": "rsi.allow_short",
    },
    "MovingAverageCrossStrategy": {
        "short_window": "moving_average_cross.short_window",
        "long_window": "moving_average_cross.long_window",
        "allow_short": "moving_average_cross.allow_short",
    },
    "RareEventScoreStrategy": {
        "zscore_threshold": "rare_event_score.zscore_threshold",
        "hold_bars": "rare_event_score.hold_bars",
        "allow_short": "rare_event_score.allow_short",
    },
}

_DEFAULT_OPTUNA: dict[str, dict[str, Any]] = {
    "PublicSampleStrategy": {
        "n_trials": 20,
        "params": {
            "decision_cadence_seconds": {"type": "int", "low": 5, "high": 120},
        },
    },
    "RsiStrategy": {
        "n_trials": 20,
        "params": {
            "rsi_period": {"type": "int", "low": 6, "high": 30},
            "oversold": {"type": "float", "low": 10.0, "high": 40.0},
            "overbought": {"type": "float", "low": 60.0, "high": 90.0},
        },
    },
    "MovingAverageCrossStrategy": {
        "n_trials": 20,
        "params": {
            "short_window": {"type": "int", "low": 5, "high": 60},
            "long_window": {"type": "int", "low": 20, "high": 200},
        },
    },
    "RareEventScoreStrategy": {
        "n_trials": 20,
        "params": {
            "zscore_threshold": {"type": "float", "low": 1.0, "high": 5.0},
            "hold_bars": {"type": "int", "low": 5, "high": 100},
        },
    },
}

_DEFAULT_GRID: dict[str, dict[str, Any]] = {
    "PublicSampleStrategy": {
        "params": {
            "decision_cadence_seconds": [10, 20, 30],
        },
    },
    "RsiStrategy": {
        "params": {
            "rsi_period": [8, 14, 21],
            "oversold": [20.0, 30.0, 40.0],
            "overbought": [60.0, 70.0, 80.0],
        },
    },
    "MovingAverageCrossStrategy": {
        "params": {
            "short_window": [10, 20, 30],
            "long_window": [40, 60, 100],
        },
    },
    "RareEventScoreStrategy": {
        "params": {
            "zscore_threshold": [1.5, 2.0, 2.5],
            "hold_bars": [10, 20, 30],
        },
    },
}


def _load_private_registry_module():
    for module_name in (
        "lumina_quant_private.strategy_registry",
        "lumina_quant_private.strategies.registry",
    ):
        try:
            return import_module(module_name)
        except Exception:
            continue
    return None


_PRIVATE_REGISTRY = _load_private_registry_module()


def _private_call(name: str, *args, **kwargs):
    if _PRIVATE_REGISTRY is None:
        return None
    fn = getattr(_PRIVATE_REGISTRY, name, None)
    if not callable(fn):
        return None
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _private_strategy_map() -> dict[str, StrategyClass]:
    private_map = _private_call("get_strategy_map")
    if isinstance(private_map, dict):
        return dict(private_map)
    return {}


def _merged_strategy_map() -> dict[str, StrategyClass]:
    merged = dict(_STRATEGY_MAP)
    merged.update(_private_strategy_map())
    return merged


def _resolve_strategy_token(strategy_name: str) -> str:
    token = str(strategy_name or "").strip()
    if token in _merged_strategy_map():
        return token
    return DEFAULT_STRATEGY_NAME


def get_strategy_map() -> dict[str, StrategyClass]:
    return _merged_strategy_map()


def get_live_strategy_map(*, include_opt_in: bool = True) -> dict[str, StrategyClass]:
    _ = include_opt_in
    private_map = _private_call("get_live_strategy_map", include_opt_in=include_opt_in)
    merged = dict(_STRATEGY_MAP)
    if isinstance(private_map, dict):
        merged.update(private_map)
    else:
        merged.update(_private_strategy_map())
    return merged


def get_strategy_names(*, include_research_only: bool = True) -> list[str]:
    _ = include_research_only
    return sorted(_merged_strategy_map().keys())


def get_live_strategy_names(*, include_opt_in: bool = True) -> list[str]:
    _ = include_opt_in
    names = _private_call("get_live_strategy_names", include_opt_in=include_opt_in)
    if isinstance(names, list) and names:
        return sorted(set(get_strategy_names()) | {str(name) for name in names})
    return sorted(get_live_strategy_map(include_opt_in=include_opt_in).keys())


def resolve_strategy_class(name: str | None, default_name: str = DEFAULT_STRATEGY_NAME) -> StrategyClass:
    merged = _merged_strategy_map()
    requested = str(name or "").strip()
    if requested in merged:
        return merged[requested]
    fallback = str(default_name or DEFAULT_STRATEGY_NAME).strip()
    if fallback in merged:
        return merged[fallback]
    return merged[DEFAULT_STRATEGY_NAME]


def get_strategy_metadata(strategy_name: str) -> dict[str, Any]:
    private_meta = _private_call("get_strategy_metadata", strategy_name)
    if isinstance(private_meta, dict) and private_meta:
        return dict(private_meta)
    token = str(strategy_name)
    return {"name": token, "tier": _STRATEGY_TIER.get(token, "live_default")}


def get_strategy_tier(strategy_name: str) -> str:
    private_tier = _private_call("get_strategy_tier", strategy_name)
    if isinstance(private_tier, str) and private_tier.strip():
        return private_tier.strip()
    return str(get_strategy_metadata(strategy_name).get("tier", "live_default"))


def get_strategy_param_schema(strategy_name: str) -> dict[str, Any]:
    private_schema = _private_call("get_strategy_param_schema", strategy_name)
    if isinstance(private_schema, dict) and private_schema:
        return deepcopy(private_schema)
    token = _resolve_strategy_token(strategy_name)
    return deepcopy(_PARAM_SCHEMAS.get(token, {}))


def get_strategy_canonical_param_names(strategy_name: str) -> dict[str, str]:
    private_names = _private_call("get_strategy_canonical_param_names", strategy_name)
    if isinstance(private_names, dict) and private_names:
        return deepcopy(private_names)
    token = _resolve_strategy_token(strategy_name)
    return deepcopy(_CANONICAL_PARAM_NAMES.get(token, {}))


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _coerce_known_param(value: Any, schema: dict[str, Any], *, default: Any) -> Any:
    kind = str(schema.get("type", "")).strip().lower()
    if kind == "int":
        try:
            coerced = int(float(value))
        except Exception:
            coerced = int(default)
        min_v = schema.get("min")
        max_v = schema.get("max")
        if isinstance(min_v, (int, float)):
            coerced = max(int(min_v), coerced)
        if isinstance(max_v, (int, float)):
            coerced = min(int(max_v), coerced)
        return coerced

    if kind == "float":
        try:
            coerced_f = float(value)
        except Exception:
            coerced_f = float(default)
        min_v = schema.get("min")
        max_v = schema.get("max")
        if isinstance(min_v, (int, float)):
            coerced_f = max(float(min_v), coerced_f)
        if isinstance(max_v, (int, float)):
            coerced_f = min(float(max_v), coerced_f)
        return coerced_f

    if kind == "bool":
        return _coerce_bool(value, bool(default))

    return value


def resolve_strategy_params(strategy_name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    private_params = _private_call("resolve_strategy_params", strategy_name, overrides)
    if isinstance(private_params, dict) and strategy_name in _private_strategy_map():
        return deepcopy(private_params)

    token = _resolve_strategy_token(strategy_name)
    resolved = deepcopy(_DEFAULT_PARAMS.get(token, {}))
    schema = _PARAM_SCHEMAS.get(token, {})

    for key, value in dict(overrides or {}).items():
        if key in schema:
            resolved[key] = _coerce_known_param(value, schema[key], default=resolved.get(key))
        else:
            resolved[key] = value

    return resolved


def get_default_strategy_params(strategy_name: str) -> dict[str, Any]:
    private_defaults = _private_call("get_default_strategy_params", strategy_name)
    if isinstance(private_defaults, dict) and strategy_name in _private_strategy_map():
        return deepcopy(private_defaults)
    token = _resolve_strategy_token(strategy_name)
    return deepcopy(_DEFAULT_PARAMS.get(token, {}))


def get_default_optuna_config(strategy_name: str) -> dict[str, Any]:
    private_optuna = _private_call("get_default_optuna_config", strategy_name)
    if isinstance(private_optuna, dict) and strategy_name in _private_strategy_map():
        return deepcopy(private_optuna)
    token = _resolve_strategy_token(strategy_name)
    return deepcopy(_DEFAULT_OPTUNA.get(token, {"n_trials": 20, "params": {}}))


def get_default_grid_config(strategy_name: str) -> dict[str, Any]:
    private_grid = _private_call("get_default_grid_config", strategy_name)
    if isinstance(private_grid, dict) and strategy_name in _private_strategy_map():
        return deepcopy(private_grid)
    token = _resolve_strategy_token(strategy_name)
    return deepcopy(_DEFAULT_GRID.get(token, {"params": {}}))


def resolve_optuna_config(strategy_name: str, override: dict[str, Any] | None = None) -> dict[str, Any]:
    private_optuna = _private_call("resolve_optuna_config", strategy_name, override)
    if isinstance(private_optuna, dict) and strategy_name in _private_strategy_map():
        return deepcopy(private_optuna)

    token = _resolve_strategy_token(strategy_name)
    cfg = get_default_optuna_config(token)
    schema_keys = set(_PARAM_SCHEMAS.get(token, {}))

    override_dict = dict(override or {})
    if "n_trials" in override_dict:
        try:
            cfg["n_trials"] = max(1, int(override_dict["n_trials"]))
        except Exception:
            pass

    override_params = override_dict.get("params")
    if isinstance(override_params, dict):
        merged_params = dict(cfg.get("params", {}))
        merged_params.update({key: value for key, value in override_params.items() if key in schema_keys})
        cfg["params"] = merged_params

    return cfg


def resolve_grid_config(strategy_name: str, override: dict[str, Any] | None = None) -> dict[str, Any]:
    private_grid = _private_call("resolve_grid_config", strategy_name, override)
    if isinstance(private_grid, dict) and strategy_name in _private_strategy_map():
        return deepcopy(private_grid)

    token = _resolve_strategy_token(strategy_name)
    cfg = get_default_grid_config(token)
    schema_keys = set(_PARAM_SCHEMAS.get(token, {}))

    override_dict = dict(override or {})
    override_params = override_dict.get("params")
    if isinstance(override_params, dict):
        merged_params = dict(cfg.get("params", {}))
        merged_params.update({key: value for key, value in override_params.items() if key in schema_keys})
        cfg["params"] = merged_params

    return cfg
