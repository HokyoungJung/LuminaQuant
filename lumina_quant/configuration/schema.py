"""Typed runtime configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SystemConfig:
    """System-level runtime settings."""

    log_level: str = "INFO"


@dataclass(slots=True)
class TradingConfig:
    """Shared trading settings."""

    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT"])
    timeframe: str = "1m"
    initial_capital: float = 10000.0
    target_allocation: float = 0.1
    min_trade_qty: float = 0.001


@dataclass(slots=True)
class RiskConfig:
    """Shared risk settings."""

    risk_per_trade: float = 0.005
    max_daily_loss_pct: float = 0.03
    max_total_margin_pct: float = 0.50
    max_symbol_exposure_pct: float = 0.25
    max_order_value: float = 5000.0
    default_stop_loss_pct: float = 0.01


@dataclass(slots=True)
class ExecutionConfig:
    """Shared execution model settings."""

    maker_fee_rate: float = 0.0002
    taker_fee_rate: float = 0.0004
    spread_rate: float = 0.0002
    slippage_rate: float = 0.0005
    funding_rate_per_8h: float = 0.0
    funding_interval_hours: int = 8
    maintenance_margin_rate: float = 0.005
    liquidation_buffer_rate: float = 0.0005
    compute_backend: str = "cpu"


@dataclass(slots=True)
class StorageConfig:
    """Storage settings for audit and exports."""

    backend: str = "sqlite"
    sqlite_path: str = "data/lumina_quant.db"
    market_data_sqlite_path: str = "data/lumina_quant.db"
    market_data_exchange: str = "binance"
    export_csv: bool = True


@dataclass(slots=True)
class BacktestRuntimeConfig:
    """Backtest-only settings."""

    start_date: str = "2024-01-01"
    end_date: str | None = None
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    annual_periods: int = 252
    risk_free_rate: float = 0.0
    random_seed: int = 42
    persist_output: bool = True
    leverage: int = 3


@dataclass(slots=True)
class LiveExchangeConfig:
    """Live exchange settings."""

    driver: str = "ccxt"
    name: str = "binance"
    market_type: str = "future"
    position_mode: str = "HEDGE"
    margin_mode: str = "isolated"
    leverage: int = 3


@dataclass(slots=True)
class LiveRuntimeConfig:
    """Live-only settings."""

    mode: str = "paper"
    require_real_enable_flag: bool = True
    poll_interval: int = 2
    order_timeout: int = 10
    heartbeat_interval_sec: int = 30
    reconciliation_interval_sec: int = 30
    exchange: LiveExchangeConfig = field(default_factory=LiveExchangeConfig)
    symbol_limits: dict[str, dict[str, float]] = field(default_factory=dict)
    api_key: str = ""
    secret_key: str = ""
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    testnet: bool | None = None
    mt5_magic: int = 234000
    mt5_deviation: int = 20


@dataclass(slots=True)
class OptimizationRuntimeConfig:
    """Optimization-only settings."""

    method: str = "OPTUNA"
    strategy: str = "RsiStrategy"
    optuna: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, Any] = field(default_factory=dict)
    walk_forward_folds: int = 3
    overfit_penalty: float = 0.5
    max_workers: int = 4
    persist_best_params: bool = False


@dataclass(slots=True)
class RuntimeConfig:
    """Full runtime configuration bundle."""

    system: SystemConfig = field(default_factory=SystemConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    backtest: BacktestRuntimeConfig = field(default_factory=BacktestRuntimeConfig)
    live: LiveRuntimeConfig = field(default_factory=LiveRuntimeConfig)
    optimization: OptimizationRuntimeConfig = field(default_factory=OptimizationRuntimeConfig)
