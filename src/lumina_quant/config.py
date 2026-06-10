"""Backward-compat shim for operational scripts (Phase 1 bridge).

The metaclass-backed runtime_access.py has been deleted.  This module now
exposes the four XxxConfig names as view objects constructed from the typed
RuntimeConfig.  No global state, no os.environ hidden bus.

Operational scripts should migrate to ``lumina_quant.configuration`` directly.
This shim will be removed in Phase 7.
"""

from __future__ import annotations

from lumina_quant.configuration import (
    BacktestConfigView,
    LiveConfigView,
    current_market_data_runtime_settings,
    get_default_runtime_config,
)
from lumina_quant.configuration.loader import load_yaml_config

# Load once at import time — consistent with old per-process behaviour.
# Scripts that need fresh values for a different LQ_CONFIG_PATH should
# call get_default_runtime_config() directly.
_rt = get_default_runtime_config()

BacktestConfig = BacktestConfigView(_rt)
LiveConfig = LiveConfigView(_rt)
# BaseConfig covers the same attribute set (SYMBOLS, TIMEFRAME, POSTGRES_DSN, …)
BaseConfig = BacktestConfigView(_rt)


class _OptimizationConfigShim:
    """Minimal shim for operational scripts that read OptimizationConfig attrs."""

    def __init__(self, opt) -> None:
        self.METHOD = str(opt.method)
        self.STRATEGY_NAME = str(opt.strategy)
        self.OPTUNA_CONFIG: dict = dict(opt.optuna)
        self.GRID_CONFIG: dict = dict(opt.grid)
        self.MAX_WORKERS = int(opt.max_workers)
        self.WALK_FORWARD_FOLDS = int(opt.walk_forward_folds)
        self.OVERFIT_PENALTY = float(opt.overfit_penalty)
        self.PERSIST_BEST_PARAMS = bool(opt.persist_best_params)
        self.VALIDATION_DAYS = int(opt.validation_days)
        self.OOS_DAYS = int(opt.oos_days)


OptimizationConfig = _OptimizationConfigShim(_rt.optimization)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load raw YAML config dict (backward compat)."""
    return load_yaml_config(config_path=config_path)


__all__ = [
    "BacktestConfig",
    "BaseConfig",
    "LiveConfig",
    "OptimizationConfig",
    "current_market_data_runtime_settings",
    "get_default_runtime_config",
    "load_config",
]
