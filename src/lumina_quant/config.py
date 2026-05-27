"""Compatibility facade for runtime config access."""

from __future__ import annotations

import importlib
import os

from lumina_quant.configuration.loader import load_yaml_config

_runtime_access = importlib.import_module("lumina_quant.configuration.runtime_access")

_runtime_access.clear_runtime_config_views()

BaseConfig = _runtime_access.BaseConfig
BacktestConfig = _runtime_access.BacktestConfig
LiveConfig = _runtime_access.LiveConfig
OptimizationConfig = _runtime_access.OptimizationConfig
clear_runtime_config_views = _runtime_access.clear_runtime_config_views
current_market_data_runtime_settings = _runtime_access.current_market_data_runtime_settings
export_runtime_dict = _runtime_access.export_runtime_dict
load_current_runtime_config = _runtime_access.load_current_runtime_config
reload_runtime_config = _runtime_access.reload_runtime_config
reset_runtime_config_cache = _runtime_access.reset_runtime_config_cache
seed_runtime_env_defaults = _runtime_access.seed_runtime_env_defaults


def load_config(config_path: str = "config.yaml") -> dict:
    """Load raw YAML config."""
    return load_yaml_config(config_path=config_path)


__all__ = [
    "BacktestConfig",
    "BaseConfig",
    "LiveConfig",
    "OptimizationConfig",
    "clear_runtime_config_views",
    "current_market_data_runtime_settings",
    "export_runtime_dict",
    "load_config",
    "load_current_runtime_config",
    "os",
    "reload_runtime_config",
    "reset_runtime_config_cache",
    "seed_runtime_env_defaults",
]
