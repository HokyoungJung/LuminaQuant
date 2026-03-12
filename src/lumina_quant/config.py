"""Typed runtime config access for backtest/live modules."""

from __future__ import annotations

import json
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
os.environ.setdefault("LQ__STORAGE__BACKEND", str(_RUNTIME.storage.backend))
os.environ.setdefault(
    "LQ__STORAGE__MARKET_DATA_PARQUET_PATH",
    str(_RUNTIME.storage.market_data_parquet_path or "data/market_parquet"),
)
os.environ.setdefault(
    "LQ__STORAGE__WAL_MAX_BYTES",
    str(int(getattr(_RUNTIME.storage, "wal_max_bytes", 268435456))),
)
os.environ.setdefault(
    "LQ__STORAGE__WAL_COMPACT_ON_THRESHOLD",
    "1" if bool(getattr(_RUNTIME.storage, "wal_compact_on_threshold", True)) else "0",
)
os.environ.setdefault(
    "LQ__STORAGE__WAL_COMPACTION_INTERVAL_SECONDS",
    str(int(getattr(_RUNTIME.storage, "wal_compaction_interval_seconds", 3600))),
)
os.environ.setdefault(
    "LQ__STORAGE__COLLECTOR_PERIODIC_ENABLED",
    "1" if bool(getattr(_RUNTIME.storage, "collector_periodic_enabled", True)) else "0",
)
os.environ.setdefault(
    "LQ__STORAGE__COLLECTOR_POLL_SECONDS",
    str(int(getattr(_RUNTIME.storage, "collector_poll_seconds", 2))),
)
os.environ.setdefault(
    "LQ__STORAGE__COLLECTOR_BOOTSTRAP_LOOKBACK_HOURS",
    str(int(getattr(_RUNTIME.storage, "collector_bootstrap_lookback_hours", 24))),
)
os.environ.setdefault(
    "LQ__STORAGE__MATERIALIZER_PERIODIC_ENABLED",
    "1" if bool(getattr(_RUNTIME.storage, "materializer_periodic_enabled", True)) else "0",
)
os.environ.setdefault(
    "LQ__STORAGE__MATERIALIZER_POLL_SECONDS",
    str(int(getattr(_RUNTIME.storage, "materializer_poll_seconds", 5))),
)
os.environ.setdefault(
    "LQ__STORAGE__MATERIALIZER_BASE_TIMEFRAME",
    str(getattr(_RUNTIME.storage, "materializer_base_timeframe", "1s") or "1s"),
)
os.environ.setdefault(
    "LQ__STORAGE__MATERIALIZER_REQUIRED_TIMEFRAMES",
    json.dumps(list(getattr(_RUNTIME.storage, "materializer_required_timeframes", ["1s"]))),
)
if str(_RUNTIME.storage.postgres_dsn or "").strip():
    os.environ.setdefault(
        str(_RUNTIME.storage.postgres_dsn_env or "LQ_POSTGRES_DSN"),
        str(_RUNTIME.storage.postgres_dsn),
    )
os.environ.setdefault(
    "LQ_GPU_MODE",
    str(getattr(_RUNTIME.execution, "gpu_mode", _RUNTIME.execution.compute_backend) or "gpu"),
)
os.environ.setdefault(
    "LQ_GPU_VRAM_GB",
    str(float(getattr(_RUNTIME.execution, "gpu_vram_gb", 0.0))),
)
os.environ.setdefault(
    "LQ__EXECUTION__GPU_MODE",
    str(getattr(_RUNTIME.execution, "gpu_mode", _RUNTIME.execution.compute_backend) or "gpu"),
)
os.environ.setdefault(
    "LQ__EXECUTION__GPU_VRAM_GB",
    str(float(getattr(_RUNTIME.execution, "gpu_vram_gb", 0.0))),
)
os.environ.setdefault(
    "LQ__BACKTEST__CHUNK_DAYS", str(int(getattr(_RUNTIME.backtest, "chunk_days", 2)))
)
os.environ.setdefault(
    "LQ__BACKTEST__CHUNK_WARMUP_BARS",
    str(int(getattr(_RUNTIME.backtest, "chunk_warmup_bars", 0))),
)
os.environ.setdefault(
    "LQ__BACKTEST__SKIP_AHEAD_ENABLED",
    "1" if bool(getattr(_RUNTIME.backtest, "skip_ahead_enabled", True)) else "0",
)
os.environ.setdefault(
    "LQ__LIVE__POLL_SECONDS",
    str(int(getattr(_RUNTIME.live, "poll_seconds", getattr(_RUNTIME.live, "poll_interval", 20)))),
)
os.environ.setdefault(
    "LQ__LIVE__WINDOW_SECONDS",
    str(int(getattr(_RUNTIME.live, "window_seconds", 20))),
)
os.environ.setdefault(
    "LQ__LIVE__MARKET_DATA_SOURCE",
    str(getattr(_RUNTIME.live, "market_data_source", "committed")),
)
os.environ.setdefault(
    "LQ__LIVE__ORDER_STATE_SOURCE",
    str(getattr(_RUNTIME.live, "order_state_source", "polling")),
)
os.environ.setdefault(
    "LQ__LIVE__SHADOW_LIVE_ENABLED",
    "1" if bool(getattr(_RUNTIME.live, "shadow_live_enabled", False)) else "0",
)
os.environ.setdefault(
    "LQ__LIVE__RECONCILIATION_POLL_FALLBACK_ENABLED",
    "1" if bool(getattr(_RUNTIME.live, "reconciliation_poll_fallback_enabled", True)) else "0",
)
os.environ.setdefault(
    "LQ__LIVE__BOOK_TICKER_ENABLED",
    "1" if bool(getattr(_RUNTIME.live, "book_ticker_enabled", False)) else "0",
)
os.environ.setdefault(
    "LQ__LIVE__STARTUP_RECONCILIATION_HARD_FAIL",
    "1" if bool(getattr(_RUNTIME.live, "startup_reconciliation_hard_fail", False)) else "0",
)
os.environ.setdefault(
    "LQ__LIVE__MATERIALIZED_STALENESS_THRESHOLD_SECONDS",
    str(int(getattr(_RUNTIME.live, "materialized_staleness_threshold_seconds", 45))),
)
os.environ.setdefault(
    "LQ__LIVE__MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS",
    str(int(getattr(_RUNTIME.live, "materialized_staleness_alert_cooldown_seconds", 60))),
)
os.environ.setdefault(
    "LQ__BACKTEST__POLL_SECONDS",
    str(int(getattr(_RUNTIME.backtest, "poll_seconds", 20))),
)
os.environ.setdefault(
    "LQ__BACKTEST__WINDOW_SECONDS",
    str(int(getattr(_RUNTIME.backtest, "window_seconds", 20))),
)
os.environ.setdefault(
    "LQ__BACKTEST__DECISION_CADENCE_SECONDS",
    str(int(getattr(_RUNTIME.backtest, "decision_cadence_seconds", 20))),
)
os.environ.setdefault(
    "LQ__MARKET_WINDOW__PARITY_V2_ENABLED",
    "1" if bool(getattr(_RUNTIME.market_window, "parity_v2_enabled", False)) else "0",
)
os.environ.setdefault(
    "LQ__MARKET_WINDOW__METRICS_LOG_PATH",
    str(
        getattr(
            _RUNTIME.market_window, "metrics_log_path", "logs/live/market_window_metrics.ndjson"
        )
        or "logs/live/market_window_metrics.ndjson"
    ),
)


class BaseConfig:
    """Shared configuration fields used by backtest and live modules."""

    LOG_LEVEL = _RUNTIME.system.log_level
    SYMBOLS = list(_RUNTIME.trading.symbols)
    TIMEFRAME = _RUNTIME.trading.timeframe
    TIMEFRAMES = list(getattr(_RUNTIME.trading, "timeframes", [])) or [str(TIMEFRAME)]
    INITIAL_CAPITAL = float(_RUNTIME.trading.initial_capital)
    TARGET_ALLOCATION = float(_RUNTIME.trading.target_allocation)
    MIN_TRADE_QTY = float(_RUNTIME.trading.min_trade_qty)

    RISK_PER_TRADE = float(_RUNTIME.risk.risk_per_trade)
    MAX_DAILY_LOSS_PCT = float(_RUNTIME.risk.max_daily_loss_pct)
    MAX_TOTAL_MARGIN_PCT = float(_RUNTIME.risk.max_total_margin_pct)
    MAX_SYMBOL_EXPOSURE_PCT = float(_RUNTIME.risk.max_symbol_exposure_pct)
    MAX_ORDER_VALUE = float(_RUNTIME.risk.max_order_value)
    DEFAULT_STOP_LOSS_PCT = float(_RUNTIME.risk.default_stop_loss_pct)
    MAX_INTRADAY_DRAWDOWN_PCT = float(_RUNTIME.risk.max_intraday_drawdown_pct)
    MAX_ROLLING_LOSS_PCT_1H = float(_RUNTIME.risk.max_rolling_loss_pct_1h)
    FREEZE_NEW_ENTRIES_ON_BREACH = bool(_RUNTIME.risk.freeze_new_entries_on_breach)
    AUTO_FLATTEN_ON_BREACH = bool(_RUNTIME.risk.auto_flatten_on_breach)

    MAKER_FEE_RATE = float(_RUNTIME.execution.maker_fee_rate)
    TAKER_FEE_RATE = float(_RUNTIME.execution.taker_fee_rate)
    SPREAD_RATE = float(_RUNTIME.execution.spread_rate)
    SLIPPAGE_RATE = float(_RUNTIME.execution.slippage_rate)
    FUNDING_RATE_PER_8H = float(_RUNTIME.execution.funding_rate_per_8h)
    FUNDING_INTERVAL_HOURS = int(_RUNTIME.execution.funding_interval_hours)
    MAINTENANCE_MARGIN_RATE = float(_RUNTIME.execution.maintenance_margin_rate)
    LIQUIDATION_BUFFER_RATE = float(_RUNTIME.execution.liquidation_buffer_rate)
    GPU_MODE = str(
        getattr(_RUNTIME.execution, "gpu_mode", _RUNTIME.execution.compute_backend) or "gpu"
    ).lower()
    GPU_VRAM_GB = float(getattr(_RUNTIME.execution, "gpu_vram_gb", 0.0))
    COMPUTE_BACKEND = GPU_MODE

    STORAGE_BACKEND = _RUNTIME.storage.backend
    STORAGE_MARKET_DATA_PARQUET_PATH = _RUNTIME.storage.market_data_parquet_path
    MARKET_DATA_PARQUET_PATH = _RUNTIME.storage.market_data_parquet_path
    MARKET_DATA_EXCHANGE = _RUNTIME.storage.market_data_exchange
    POSTGRES_DSN_ENV = str(getattr(_RUNTIME.storage, "postgres_dsn_env", "LQ_POSTGRES_DSN"))
    POSTGRES_DSN = str(getattr(_RUNTIME.storage, "postgres_dsn", "") or "")
    STORAGE_EXPORT_CSV = bool(_RUNTIME.storage.export_csv)
    WAL_MAX_BYTES = int(getattr(_RUNTIME.storage, "wal_max_bytes", 268435456))
    WAL_COMPACT_ON_THRESHOLD = bool(getattr(_RUNTIME.storage, "wal_compact_on_threshold", True))
    WAL_COMPACTION_INTERVAL_SECONDS = int(
        getattr(_RUNTIME.storage, "wal_compaction_interval_seconds", 3600)
    )
    COLLECTOR_PERIODIC_ENABLED = bool(getattr(_RUNTIME.storage, "collector_periodic_enabled", True))
    COLLECTOR_POLL_SECONDS = int(getattr(_RUNTIME.storage, "collector_poll_seconds", 2))
    COLLECTOR_BOOTSTRAP_LOOKBACK_HOURS = int(
        getattr(_RUNTIME.storage, "collector_bootstrap_lookback_hours", 24)
    )
    MATERIALIZER_PERIODIC_ENABLED = bool(
        getattr(_RUNTIME.storage, "materializer_periodic_enabled", True)
    )
    MATERIALIZER_POLL_SECONDS = int(getattr(_RUNTIME.storage, "materializer_poll_seconds", 5))
    MATERIALIZER_BASE_TIMEFRAME = str(
        getattr(_RUNTIME.storage, "materializer_base_timeframe", "1s") or "1s"
    ).lower()
    MATERIALIZER_REQUIRED_TIMEFRAMES = list(
        getattr(_RUNTIME.storage, "materializer_required_timeframes", ["1s"])
    ) or ["1s"]

    MARKET_WINDOW_PARITY_V2_ENABLED = bool(
        getattr(_RUNTIME.market_window, "parity_v2_enabled", False)
    )
    MARKET_WINDOW_METRICS_LOG_PATH = str(
        getattr(
            _RUNTIME.market_window, "metrics_log_path", "logs/live/market_window_metrics.ndjson"
        )
        or "logs/live/market_window_metrics.ndjson"
    )


class BacktestConfig(BaseConfig):
    """Backtest configuration access."""

    START_DATE = _RUNTIME.backtest.start_date
    END_DATE = _RUNTIME.backtest.end_date
    MODE = str(getattr(_RUNTIME.backtest, "mode", "windowed") or "windowed")
    COMMISSION_RATE = float(_RUNTIME.backtest.commission_rate)
    SLIPPAGE_RATE = float(_RUNTIME.backtest.slippage_rate)
    ANNUAL_PERIODS = int(_RUNTIME.backtest.annual_periods)
    RISK_FREE_RATE = float(_RUNTIME.backtest.risk_free_rate)
    RANDOM_SEED = int(_RUNTIME.backtest.random_seed)
    PERSIST_OUTPUT = bool(_RUNTIME.backtest.persist_output)
    LEVERAGE = int(_RUNTIME.backtest.leverage)
    POLL_SECONDS = int(getattr(_RUNTIME.backtest, "poll_seconds", 20))
    WINDOW_SECONDS = int(getattr(_RUNTIME.backtest, "window_seconds", 20))
    DECISION_CADENCE_SECONDS = int(getattr(_RUNTIME.backtest, "decision_cadence_seconds", 20))
    BACKTEST_POLL_SECONDS = POLL_SECONDS
    BACKTEST_WINDOW_SECONDS = WINDOW_SECONDS
    BACKTEST_DECISION_SECONDS = DECISION_CADENCE_SECONDS
    CHUNK_DAYS = int(getattr(_RUNTIME.backtest, "chunk_days", 2))
    CHUNK_WARMUP_BARS = int(getattr(_RUNTIME.backtest, "chunk_warmup_bars", 0))
    SKIP_AHEAD_ENABLED = bool(getattr(_RUNTIME.backtest, "skip_ahead_enabled", True))


class LiveConfig(BaseConfig):
    """Live configuration access with runtime validation helpers."""

    BINANCE_API_KEY = _RUNTIME.live.api_key
    BINANCE_SECRET_KEY = _RUNTIME.live.secret_key
    TELEGRAM_BOT_TOKEN = _RUNTIME.live.telegram_bot_token
    TELEGRAM_CHAT_ID = _RUNTIME.live.telegram_chat_id

    MODE = str(_RUNTIME.live.mode).strip().lower()
    MARKET_DATA_SOURCE = (
        str(getattr(_RUNTIME.live, "market_data_source", "committed")).strip().lower()
    )
    ORDER_STATE_SOURCE = (
        str(getattr(_RUNTIME.live, "order_state_source", "polling")).strip().lower()
    )
    SHADOW_LIVE_ENABLED = bool(getattr(_RUNTIME.live, "shadow_live_enabled", False))
    RECONCILIATION_POLL_FALLBACK_ENABLED = bool(
        getattr(_RUNTIME.live, "reconciliation_poll_fallback_enabled", True)
    )
    BOOK_TICKER_ENABLED = bool(getattr(_RUNTIME.live, "book_ticker_enabled", False))
    STARTUP_RECONCILIATION_HARD_FAIL = bool(
        getattr(_RUNTIME.live, "startup_reconciliation_hard_fail", False)
    )

    IS_TESTNET = MODE != "real"
    REQUIRE_REAL_ENABLE_FLAG = bool(_RUNTIME.live.require_real_enable_flag)
    POLL_SECONDS = int(getattr(_RUNTIME.live, "poll_seconds", _RUNTIME.live.poll_interval))
    POLL_INTERVAL = POLL_SECONDS
    LIVE_POLL_SECONDS = POLL_SECONDS
    WINDOW_SECONDS = int(getattr(_RUNTIME.live, "window_seconds", 20))
    INGEST_WINDOW_SECONDS = WINDOW_SECONDS
    DECISION_CADENCE_SECONDS = int(getattr(_RUNTIME.live, "decision_cadence_seconds", 20))
    MATERIALIZED_STALENESS_THRESHOLD_SECONDS = int(
        getattr(_RUNTIME.live, "materialized_staleness_threshold_seconds", 45)
    )
    MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS = int(
        getattr(_RUNTIME.live, "materialized_staleness_alert_cooldown_seconds", 60)
    )
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
    MT5_BRIDGE_PYTHON = str(getattr(_RUNTIME.live, "mt5_bridge_python", "") or "")
    MT5_BRIDGE_SCRIPT = str(
        getattr(_RUNTIME.live, "mt5_bridge_script", "scripts/mt5_bridge_worker.py")
        or "scripts/mt5_bridge_worker.py"
    )
    MT5_BRIDGE_USE_WSLPATH = bool(getattr(_RUNTIME.live, "mt5_bridge_use_wslpath", True))

    @classmethod
    def _as_runtime(cls):
        runtime = load_runtime_config(config_path=os.getenv("LQ_CONFIG_PATH", "config.yaml"))
        runtime.live.mode = cls.MODE
        runtime.live.market_data_source = str(cls.MARKET_DATA_SOURCE)
        runtime.live.order_state_source = str(cls.ORDER_STATE_SOURCE)
        runtime.live.shadow_live_enabled = bool(cls.SHADOW_LIVE_ENABLED)
        runtime.live.reconciliation_poll_fallback_enabled = bool(
            cls.RECONCILIATION_POLL_FALLBACK_ENABLED
        )
        runtime.live.book_ticker_enabled = bool(cls.BOOK_TICKER_ENABLED)
        runtime.live.startup_reconciliation_hard_fail = bool(cls.STARTUP_RECONCILIATION_HARD_FAIL)
        runtime.live.require_real_enable_flag = cls.REQUIRE_REAL_ENABLE_FLAG
        runtime.live.materialized_staleness_threshold_seconds = int(
            cls.MATERIALIZED_STALENESS_THRESHOLD_SECONDS
        )
        runtime.live.materialized_staleness_alert_cooldown_seconds = int(
            cls.MATERIALIZED_STALENESS_ALERT_COOLDOWN_SECONDS
        )
        runtime.live.api_key = cls.BINANCE_API_KEY
        runtime.live.secret_key = cls.BINANCE_SECRET_KEY
        runtime.live.exchange.driver = str(cls.EXCHANGE["driver"])
        runtime.live.exchange.name = str(cls.EXCHANGE["name"])
        runtime.live.exchange.market_type = str(cls.EXCHANGE["market_type"])
        runtime.live.exchange.position_mode = str(cls.EXCHANGE["position_mode"])
        runtime.live.exchange.margin_mode = str(cls.EXCHANGE["margin_mode"])
        runtime.live.exchange.leverage = int(cls.EXCHANGE["leverage"])
        runtime.live.mt5_bridge_python = str(cls.MT5_BRIDGE_PYTHON)
        runtime.live.mt5_bridge_script = str(cls.MT5_BRIDGE_SCRIPT)
        runtime.live.mt5_bridge_use_wslpath = bool(cls.MT5_BRIDGE_USE_WSLPATH)
        runtime.storage.collector_periodic_enabled = bool(cls.COLLECTOR_PERIODIC_ENABLED)
        runtime.storage.collector_poll_seconds = int(cls.COLLECTOR_POLL_SECONDS)
        runtime.storage.collector_bootstrap_lookback_hours = int(
            cls.COLLECTOR_BOOTSTRAP_LOOKBACK_HOURS
        )
        runtime.storage.materializer_periodic_enabled = bool(cls.MATERIALIZER_PERIODIC_ENABLED)
        runtime.storage.materializer_poll_seconds = int(cls.MATERIALIZER_POLL_SECONDS)
        runtime.storage.materializer_base_timeframe = str(cls.MATERIALIZER_BASE_TIMEFRAME)
        runtime.storage.materializer_required_timeframes = list(
            cls.MATERIALIZER_REQUIRED_TIMEFRAMES
        )
        runtime.market_window.parity_v2_enabled = bool(cls.MARKET_WINDOW_PARITY_V2_ENABLED)
        runtime.market_window.metrics_log_path = str(cls.MARKET_WINDOW_METRICS_LOG_PATH)
        runtime.trading.symbols = list(cls.SYMBOLS)
        runtime.trading.timeframe = str(cls.TIMEFRAME)
        runtime.trading.timeframes = list(getattr(cls, "TIMEFRAMES", [cls.TIMEFRAME]))
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
    VALIDATION_DAYS = int(getattr(_RUNTIME.optimization, "validation_days", 30))
    OOS_DAYS = int(getattr(_RUNTIME.optimization, "oos_days", 30))


def export_runtime_dict() -> dict:
    """Export the loaded typed runtime as a plain dictionary."""
    return asdict(_RUNTIME)
