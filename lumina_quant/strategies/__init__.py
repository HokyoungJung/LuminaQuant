"""Public strategy registry exports for CLI and dashboard modules."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DEFAULT_STRATEGY_NAME",
    "get_default_grid_config",
    "get_default_optuna_config",
    "get_default_strategy_params",
    "get_live_strategy_map",
    "get_live_strategy_names",
    "get_strategy_canonical_param_names",
    "get_strategy_map",
    "get_strategy_metadata",
    "get_strategy_names",
    "get_strategy_param_schema",
    "get_strategy_tier",
    "resolve_grid_config",
    "resolve_optuna_config",
    "resolve_strategy_class",
    "resolve_strategy_params",
]

_REGISTRY_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    """Lazily resolve registry exports to avoid package-level import cycles."""
    if name in _REGISTRY_EXPORTS:
        from . import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy registry exports in interactive/module introspection."""
    return sorted(set(globals()) | _REGISTRY_EXPORTS)
