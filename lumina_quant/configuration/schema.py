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
    max_intraday_drawdown_pct: float = 0.03
    max_rolling_loss_pct_1h: float = 0.05
    freeze_new_entries_on_breach: bool = True
    auto_flatten_on_breach: bool = False


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
    compute_backend: str = "auto"


@dataclass(slots=True)
class StorageConfig:
    """Storage settings for audit and exports."""

    backend: str = "parquet-postgres"
    postgres_dsn: str = ""
    postgres_dsn_env: str = "LQ_POSTGRES_DSN"
    market_data_parquet_path: str = "data/market_parquet"
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
    mt5_bridge_python: str = ""
    mt5_bridge_script: str = "scripts/mt5_bridge_worker.py"
    mt5_bridge_use_wslpath: bool = True


@dataclass(slots=True)
class OptimizationRuntimeConfig:
    """Optimization-only settings."""

    method: str = "OPTUNA"
    strategy: str = "RsiStrategy"
    optuna: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, Any] = field(default_factory=dict)
    walk_forward_folds: int = 3
    overfit_penalty: float = 0.5
    max_workers: int = 1
    persist_best_params: bool = False


@dataclass(slots=True)
class PromotionGateConfig:
    """Promotion gate defaults and strategy-specific override profiles."""

    days: int = 14
    max_order_rejects: int = 0
    max_order_timeouts: int = 0
    max_reconciliation_alerts: int = 0
    max_critical_errors: int = 0
    require_alpha_card: bool = False
    strategy_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)


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
    promotion_gate: PromotionGateConfig = field(default_factory=PromotionGateConfig)
