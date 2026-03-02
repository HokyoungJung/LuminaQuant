"""Public-safe strategy registry with sample-only entries."""

from __future__ import annotations

from typing import Any

from lumina_quant.strategy import Strategy

from .sample_public_strategy import PublicSampleStrategy

StrategyClass = type[Strategy]

DEFAULT_STRATEGY_NAME = "PublicSampleStrategy"

_STRATEGY_MAP: dict[str, StrategyClass] = {
    "PublicSampleStrategy": PublicSampleStrategy,
    # Compatibility aliases for legacy defaults in run_live/run_backtest
    "RsiStrategy": PublicSampleStrategy,
    "MovingAverageCrossStrategy": PublicSampleStrategy,
}

_STRATEGY_TIER: dict[str, str] = dict.fromkeys(_STRATEGY_MAP, "live_default")


def get_strategy_map() -> dict[str, StrategyClass]:
    return dict(_STRATEGY_MAP)


def get_live_strategy_map(*, include_opt_in: bool = True) -> dict[str, StrategyClass]:
    _ = include_opt_in
    return dict(_STRATEGY_MAP)


def get_strategy_names(*, include_research_only: bool = True) -> list[str]:
    _ = include_research_only
    return sorted(_STRATEGY_MAP.keys())


def get_live_strategy_names(*, include_opt_in: bool = True) -> list[str]:
    _ = include_opt_in
    return sorted(_STRATEGY_MAP.keys())


def resolve_strategy_class(name: str | None, default_name: str = DEFAULT_STRATEGY_NAME) -> StrategyClass:
    requested = str(name or "").strip()
    if requested in _STRATEGY_MAP:
        return _STRATEGY_MAP[requested]
    fallback = str(default_name or DEFAULT_STRATEGY_NAME).strip()
    if fallback in _STRATEGY_MAP:
        return _STRATEGY_MAP[fallback]
    return _STRATEGY_MAP[DEFAULT_STRATEGY_NAME]


def get_strategy_metadata(strategy_name: str) -> dict[str, Any]:
    token = str(strategy_name)
    return {"name": token, "tier": _STRATEGY_TIER.get(token, "live_default")}


def get_strategy_tier(strategy_name: str) -> str:
    return str(get_strategy_metadata(strategy_name).get("tier", "live_default"))


def get_strategy_param_schema(strategy_name: str) -> dict[str, Any]:
    _ = strategy_name
    return {}


def get_strategy_canonical_param_names(strategy_name: str) -> dict[str, str]:
    _ = strategy_name
    return {}


def resolve_strategy_params(strategy_name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    _ = strategy_name
    return dict(overrides or {})


def get_default_strategy_params(strategy_name: str) -> dict[str, Any]:
    _ = strategy_name
    return {}


def get_default_optuna_config(strategy_name: str) -> dict[str, Any]:
    _ = strategy_name
    return {"n_trials": 20, "params": {}}


def get_default_grid_config(strategy_name: str) -> dict[str, Any]:
    _ = strategy_name
    return {"params": {}}


def resolve_optuna_config(strategy_name: str, override: dict[str, Any] | None = None) -> dict[str, Any]:
    _ = strategy_name
    cfg = get_default_optuna_config(strategy_name)
    if isinstance(override, dict):
        cfg.update(override)
    return cfg


def resolve_grid_config(strategy_name: str, override: dict[str, Any] | None = None) -> dict[str, Any]:
    _ = strategy_name
    cfg = get_default_grid_config(strategy_name)
    if isinstance(override, dict):
        cfg.update(override)
    return cfg
