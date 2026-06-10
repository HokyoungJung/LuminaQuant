"""Public strategy registry exports for CLI and dashboard modules.

Phase 1: replaced lazy ``__getattr__`` delegation with eager imports.
Single-source registry: ``lumina_quant.core.plugin_registry`` (decorator-based).
Operational strategy look-up functions are still re-exported from
``lumina_quant.strategies.registry`` for backward compat with CLI entry points.
"""

from __future__ import annotations

# Submodule — makes strategies.registry accessible as a module attribute
# via the normal sub-package protocol, no lazy indirection needed.
from . import registry

from lumina_quant.strategies.registry import (
    DEFAULT_STRATEGY_NAME,
    get_default_grid_config,
    get_default_optuna_config,
    get_default_strategy_params,
    get_live_strategy_map,
    get_live_strategy_names,
    get_strategy_canonical_param_names,
    get_strategy_map,
    get_strategy_metadata,
    get_strategy_names,
    get_strategy_param_schema,
    get_strategy_tier,
    resolve_grid_config,
    resolve_optuna_config,
    resolve_strategy_class,
    resolve_strategy_params,
)
from lumina_quant.core.plugin_registry import GLOBAL_REGISTRY, register

__all__ = [
    "DEFAULT_STRATEGY_NAME",
    "GLOBAL_REGISTRY",
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
    "register",
    "registry",
    "resolve_grid_config",
    "resolve_optuna_config",
    "resolve_strategy_class",
    "resolve_strategy_params",
]
