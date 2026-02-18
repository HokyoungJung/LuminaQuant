"""Typed runtime config access for backtest/live modules."""

from __future__ import annotations

import os
import re
from dataclasses import asdict
from typing import ClassVar

from lumina_quant.configuration.loader import load_runtime_config, load_yaml_config
from lumina_quant.configuration.validate import validate_runtime_config


def load_config(config_path: str = "config.yaml") -> dict:
    """Load raw YAML config."""
    return load_yaml_config(config_path=config_path)


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


_CONFIG_PATH = os.getenv("LQ_CONFIG_PATH", "config.yaml")
_RUNTIME = load_runtime_config(config_path=_CONFIG_PATH)


class BaseConfig:
    """Shared configuration fields used by backtest and live modules."""

    LOG_LEVEL = _RUNTIME.system.log_level
    SYMBOLS = list(_RUNTIME.trading.symbols)
    TIMEFRAME = _RUNTIME.trading.timeframe
    INITIAL_CAPITAL = float(_RUNTIME.trading.initial_capital)
    TARGET_ALLOCATION = float(_RUNTIME.trading.target_allocation)
    MIN_TRADE_QTY = float(_RUNTIME.trading.min_trade_qty)

    RISK_PER_TRADE = float(_RUNTIME.risk.risk_per_trade)
    MAX_DAILY_LOSS_PCT = float(_RUNTIME.risk.max_daily_loss_pct)
    MAX_TOTAL_MARGIN_PCT = float(_RUNTIME.risk.max_total_margin_pct)
    MAX_SYMBOL_EXPOSURE_PCT = float(_RUNTIME.risk.max_symbol_exposure_pct)
    MAX_ORDER_VALUE = float(_RUNTIME.risk.max_order_value)
    DEFAULT_STOP_LOSS_PCT = float(_RUNTIME.risk.default_stop_loss_pct)

    MAKER_FEE_RATE = float(_RUNTIME.execution.maker_fee_rate)
    TAKER_FEE_RATE = float(_RUNTIME.execution.taker_fee_rate)
    SPREAD_RATE = float(_RUNTIME.execution.spread_rate)
    SLIPPAGE_RATE = float(_RUNTIME.execution.slippage_rate)
    FUNDING_RATE_PER_8H = float(_RUNTIME.execution.funding_rate_per_8h)
    FUNDING_INTERVAL_HOURS = int(_RUNTIME.execution.funding_interval_hours)
    MAINTENANCE_MARGIN_RATE = float(_RUNTIME.execution.maintenance_margin_rate)
    LIQUIDATION_BUFFER_RATE = float(_RUNTIME.execution.liquidation_buffer_rate)
    COMPUTE_BACKEND = str(_RUNTIME.execution.compute_backend).lower()

    STORAGE_BACKEND = _RUNTIME.storage.backend
    STORAGE_SQLITE_PATH = _RUNTIME.storage.sqlite_path
    MARKET_DATA_SQLITE_PATH = _RUNTIME.storage.market_data_sqlite_path
    MARKET_DATA_EXCHANGE = _RUNTIME.storage.market_data_exchange
    STORAGE_EXPORT_CSV = bool(_RUNTIME.storage.export_csv)


class BacktestConfig(BaseConfig):
    """Backtest configuration access."""

    START_DATE = _RUNTIME.backtest.start_date
    END_DATE = _RUNTIME.backtest.end_date
    COMMISSION_RATE = float(_RUNTIME.backtest.commission_rate)
    SLIPPAGE_RATE = float(_RUNTIME.backtest.slippage_rate)
    ANNUAL_PERIODS = int(_RUNTIME.backtest.annual_periods)
    RISK_FREE_RATE = float(_RUNTIME.backtest.risk_free_rate)
    RANDOM_SEED = int(_RUNTIME.backtest.random_seed)
    PERSIST_OUTPUT = bool(_RUNTIME.backtest.persist_output)
    LEVERAGE = int(_RUNTIME.backtest.leverage)


class LiveConfig(BaseConfig):
    """Live configuration access with runtime validation helpers."""

    BINANCE_API_KEY = _RUNTIME.live.api_key
    BINANCE_SECRET_KEY = _RUNTIME.live.secret_key
    TELEGRAM_BOT_TOKEN = _RUNTIME.live.telegram_bot_token
    TELEGRAM_CHAT_ID = _RUNTIME.live.telegram_chat_id

    MODE = str(_RUNTIME.live.mode).strip().lower()

    IS_TESTNET = MODE != "real"
    REQUIRE_REAL_ENABLE_FLAG = bool(_RUNTIME.live.require_real_enable_flag)
    POLL_INTERVAL = int(_RUNTIME.live.poll_interval)
    ORDER_TIMEOUT = int(_RUNTIME.live.order_timeout)
    HEARTBEAT_INTERVAL_SEC = int(_RUNTIME.live.heartbeat_interval_sec)
    RECONCILIATION_INTERVAL_SEC = int(_RUNTIME.live.reconciliation_interval_sec)

    EXCHANGE: ClassVar[dict[str, str | int]] = {
        "driver": str(_RUNTIME.live.exchange.driver).lower(),
        "name": str(_RUNTIME.live.exchange.name).lower(),
        "market_type": str(_RUNTIME.live.exchange.market_type).lower(),
        "position_mode": str(_RUNTIME.live.exchange.position_mode).upper(),
        "margin_mode": str(_RUNTIME.live.exchange.margin_mode).lower(),
        "leverage": int(_RUNTIME.live.exchange.leverage),
    }
    EXCHANGE_ID = str(EXCHANGE["name"])
    MARKET_TYPE = str(EXCHANGE["market_type"])
    POSITION_MODE = str(EXCHANGE["position_mode"])
    MARGIN_MODE = str(EXCHANGE["margin_mode"])
    LEVERAGE = int(EXCHANGE["leverage"])
    SYMBOL_LIMITS = dict(_RUNTIME.live.symbol_limits)
    MT5_MAGIC = int(_RUNTIME.live.mt5_magic)
    MT5_DEVIATION = int(_RUNTIME.live.mt5_deviation)

    @classmethod
    def _as_runtime(cls):
        runtime = load_runtime_config(config_path=os.getenv("LQ_CONFIG_PATH", "config.yaml"))
        runtime.live.mode = cls.MODE
        runtime.live.require_real_enable_flag = cls.REQUIRE_REAL_ENABLE_FLAG
        runtime.live.api_key = cls.BINANCE_API_KEY
        runtime.live.secret_key = cls.BINANCE_SECRET_KEY
        runtime.live.exchange.driver = str(cls.EXCHANGE["driver"])
        runtime.live.exchange.name = str(cls.EXCHANGE["name"])
        runtime.live.exchange.market_type = str(cls.EXCHANGE["market_type"])
        runtime.live.exchange.position_mode = str(cls.EXCHANGE["position_mode"])
        runtime.live.exchange.margin_mode = str(cls.EXCHANGE["margin_mode"])
        runtime.live.exchange.leverage = int(cls.EXCHANGE["leverage"])
        runtime.trading.symbols = list(cls.SYMBOLS)
        runtime.risk.max_daily_loss_pct = cls.MAX_DAILY_LOSS_PCT
        return runtime

    @classmethod
    def validate(cls):
        """Validate live config and enforce real-trading safety flag."""
        runtime = cls._as_runtime()
        validate_runtime_config(runtime, for_live=True)

        symbol_re = re.compile(r"^[A-Z0-9]+/[A-Z0-9]+$")
        for symbol in cls.SYMBOLS:
            if not symbol_re.match(symbol):
                raise ValueError(
                    f"Invalid symbol format '{symbol}'. Expected format like BTC/USDT."
                )

        if cls.MODE == "real" and cls.REQUIRE_REAL_ENABLE_FLAG:
            real_flag = os.getenv("LUMINA_ENABLE_LIVE_REAL", "")
            if not _as_bool(real_flag, False):
                raise ValueError(
                    "Real trading is blocked by default. "
                    "Set LUMINA_ENABLE_LIVE_REAL=true to allow live real mode."
                )


class OptimizationConfig:
    """Optimization configuration access."""

    METHOD = _RUNTIME.optimization.method
    STRATEGY_NAME = _RUNTIME.optimization.strategy
    OPTUNA_CONFIG = dict(_RUNTIME.optimization.optuna)
    GRID_CONFIG = dict(_RUNTIME.optimization.grid)
    WALK_FORWARD_FOLDS = int(_RUNTIME.optimization.walk_forward_folds)
    OVERFIT_PENALTY = float(_RUNTIME.optimization.overfit_penalty)
    MAX_WORKERS = int(_RUNTIME.optimization.max_workers)
    PERSIST_BEST_PARAMS = bool(_RUNTIME.optimization.persist_best_params)


def export_runtime_dict() -> dict:
    """Export the loaded typed runtime as a plain dictionary."""
    return asdict(_RUNTIME)
