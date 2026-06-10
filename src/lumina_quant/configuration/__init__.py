"""Typed configuration API — single entry point for all configuration access.

Public API
----------
load_runtime_config(config_path)  — load YAML + env overrides → RuntimeConfig
build_runtime_config(data, env)   — build from raw dict + env mapping
load_yaml_config(config_path)     — raw YAML dict (pre-overrides)
validate_runtime_config(rt, ...)  — assert invariants on a RuntimeConfig
get_default_runtime_config()      — convenience: load from LQ_CONFIG_PATH env var
load_experiment_config(path)      — load an ExperimentConfig bundle

Schema types re-exported so callers never need to import from sub-modules:
  RuntimeConfig, BacktestRuntimeConfig, LiveRuntimeConfig, OptimizationRuntimeConfig,
  StorageConfig, TradingConfig, RiskConfig, ExecutionConfig, SystemConfig,
  MarketWindowConfig, MemoryConfig, ValidationConfig, PromotionGateConfig,
  BacktestExternalConfig, LiveExchangeConfig, LiveExternalConfig, LivePolymarketConfig

Engine views (Phase 1 bridge — deletion-gated):
  BacktestConfigView — uppercase-attr bag for Backtest / Portfolio / ExecutionSim
                       DELETION-GATE: Phase 4
  LiveConfigView     — uppercase-attr bag for LiveTrader / RiskManager
                       DELETION-GATE: Phase 5
"""

from __future__ import annotations

import os

from lumina_quant.configuration.views import BacktestConfigView, LiveConfigView
from lumina_quant.configuration.loader import (
    build_runtime_config,
    load_runtime_config,
    load_yaml_config,
)
from lumina_quant.configuration.schema import (
    BacktestExternalConfig,
    BacktestRuntimeConfig,
    ExecutionConfig,
    LiveExchangeConfig,
    LiveExternalConfig,
    LivePolymarketConfig,
    LiveRuntimeConfig,
    MarketWindowConfig,
    MemoryConfig,
    OptimizationRuntimeConfig,
    PromotionGateConfig,
    RiskConfig,
    RuntimeConfig,
    StorageConfig,
    SystemConfig,
    TradingConfig,
    ValidationConfig,
)
from lumina_quant.configuration.validate import validate_runtime_config

_DEFAULT_CONFIG_PATH = "config.yaml"


def get_default_runtime_config() -> RuntimeConfig:
    """Load RuntimeConfig from LQ_CONFIG_PATH env var (default: config.yaml).

    Non-CLI code that needs config without an explicit path should call this.
    Each call performs a fresh load — use load_runtime_config() and cache the
    result when called in a hot path.

    Returns schema defaults when the config file is absent (FileNotFoundError)
    so library code works without a config.yaml present.
    """
    path = os.environ.get("LQ_CONFIG_PATH", _DEFAULT_CONFIG_PATH)
    try:
        return load_runtime_config(path)
    except FileNotFoundError:
        return RuntimeConfig()


def current_market_data_runtime_settings() -> dict[str, object]:
    """Return current market data settings as a plain dict.

    Replacement for the old runtime_access.current_market_data_runtime_settings().
    Reads the YAML directly to honour any section-level overrides before env
    normalisation, exactly as the old implementation did.
    """
    path = os.environ.get("LQ_CONFIG_PATH", _DEFAULT_CONFIG_PATH)
    runtime = load_runtime_config(path)
    try:
        raw = load_yaml_config(config_path=path)
    except Exception:
        raw = {}
    trading_raw = raw.get("trading", {}) if isinstance(raw.get("trading"), dict) else {}
    storage_raw = raw.get("storage", {}) if isinstance(raw.get("storage"), dict) else {}
    return {
        "symbols": list(
            trading_raw.get("symbols") or runtime.trading.symbols
        ),
        "market_data_parquet_path": str(
            storage_raw.get("market_data_parquet_path")
            or runtime.storage.market_data_parquet_path
            or "data/market_parquet"
        ),
        "market_data_exchange": str(
            storage_raw.get("market_data_exchange")
            or runtime.storage.market_data_exchange
            or "binance"
        ),
    }


# Attempt to expose ExperimentConfig/load_experiment_config from the optional
# experiment_loader sub-module (not imported by all distributions).
try:
    from lumina_quant.configuration.experiment_loader import (  # noqa: F401
        ExperimentConfig,
        load_experiment_config,
    )
except ImportError:
    pass


__all__ = [
    # Core loader functions
    "build_runtime_config",
    "get_default_runtime_config",
    "current_market_data_runtime_settings",
    "load_runtime_config",
    "load_yaml_config",
    "validate_runtime_config",
    # Engine compat views
    "BacktestConfigView",
    "LiveConfigView",
    # Schema types
    "BacktestExternalConfig",
    "BacktestRuntimeConfig",
    "ExecutionConfig",
    "LiveExchangeConfig",
    "LiveExternalConfig",
    "LivePolymarketConfig",
    "LiveRuntimeConfig",
    "MarketWindowConfig",
    "MemoryConfig",
    "OptimizationRuntimeConfig",
    "PromotionGateConfig",
    "RiskConfig",
    "RuntimeConfig",
    "StorageConfig",
    "SystemConfig",
    "TradingConfig",
    "ValidationConfig",
]
