import os
import yaml
import warnings
import re
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (for API keys)
load_dotenv()


def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.
    """
    # Find config file relative to the project root or this file
    # Assuming config.py is in lumina_quant/ and config.yaml is in project root
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / config_path

    if not path.exists():
        # Fallback to looking in current directory
        path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.absolute()}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            print(f"Error parsing config.yaml: {e}")
            return {}


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


# Load the config once
_CONFIG_DATA = load_config()


class BaseConfig:
    """
    Base configuration loading from YAML.
    """

    _c = _CONFIG_DATA

    # System
    LOG_LEVEL = _c.get("system", {}).get("log_level", "INFO")

    # Trading (Shared)
    _t = _c.get("trading", {})
    SYMBOLS = _t.get("symbols", ["BTC/USDT"])
    TIMEFRAME = _t.get("timeframe", "1m")
    INITIAL_CAPITAL = _as_float(_t.get("initial_capital", 10000.0), 10000.0)
    TARGET_ALLOCATION = _as_float(_t.get("target_allocation", 0.1), 0.1)
    MIN_TRADE_QTY = _as_float(_t.get("min_trade_qty", 0.001), 0.001)

    # New Risk / Execution / Storage blocks (with backward compatibility)
    _risk = _c.get("risk", {})
    RISK_PER_TRADE = _as_float(_risk.get("risk_per_trade", 0.005), 0.005)
    MAX_DAILY_LOSS_PCT = _as_float(_risk.get("max_daily_loss_pct", 0.03), 0.03)
    MAX_TOTAL_MARGIN_PCT = _as_float(
        _risk.get("max_total_margin_pct", 0.50), 0.50
    )
    MAX_SYMBOL_EXPOSURE_PCT = _as_float(
        _risk.get("max_symbol_exposure_pct", 0.25), 0.25
    )
    MAX_ORDER_VALUE = _as_float(_risk.get("max_order_value", 5000.0), 5000.0)
    DEFAULT_STOP_LOSS_PCT = _as_float(_risk.get("default_stop_loss_pct", 0.01), 0.01)

    _exec = _c.get("execution", {})
    MAKER_FEE_RATE = _as_float(_exec.get("maker_fee_rate", 0.0002), 0.0002)
    TAKER_FEE_RATE = _as_float(_exec.get("taker_fee_rate", 0.0004), 0.0004)
    SPREAD_RATE = _as_float(_exec.get("spread_rate", 0.0002), 0.0002)
    SLIPPAGE_RATE = _as_float(_exec.get("slippage_rate", 0.0005), 0.0005)
    FUNDING_RATE_PER_8H = _as_float(_exec.get("funding_rate_per_8h", 0.0), 0.0)
    FUNDING_INTERVAL_HOURS = _as_int(_exec.get("funding_interval_hours", 8), 8)
    MAINTENANCE_MARGIN_RATE = _as_float(
        _exec.get("maintenance_margin_rate", 0.005), 0.005
    )
    LIQUIDATION_BUFFER_RATE = _as_float(
        _exec.get("liquidation_buffer_rate", 0.0005), 0.0005
    )

    _storage = _c.get("storage", {})
    STORAGE_BACKEND = _storage.get("backend", "sqlite")
    STORAGE_SQLITE_PATH = _storage.get("sqlite_path", "logs/lumina_quant.db")
    STORAGE_EXPORT_CSV = _as_bool(_storage.get("export_csv", True), True)


class BacktestConfig(BaseConfig):
    """
    Configuration for Backtesting.
    """

    _b = _CONFIG_DATA.get("backtest", {})

    START_DATE = _b.get("start_date", "2024-01-01")
    # Handle None for end_date safely
    END_DATE = _b.get("end_date")

    COMMISSION_RATE = _as_float(_b.get("commission_rate", 0.001), 0.001)
    SLIPPAGE_RATE = _as_float(_b.get("slippage_rate", BaseConfig.SLIPPAGE_RATE), 0.0005)
    ANNUAL_PERIODS = _as_int(_b.get("annual_periods", 252), 252)
    RISK_FREE_RATE = 0.0  # Optional, default to 0
    RANDOM_SEED = _as_int(_b.get("random_seed", 42), 42)
    PERSIST_OUTPUT = _as_bool(_b.get("persist_output", True), True)
    LEVERAGE = _as_int(_b.get("leverage", 3), 3)


class LiveConfig(BaseConfig):
    """
    Configuration for Live Trading.
    """

    _l = _CONFIG_DATA.get("live", {})
    _exchange = _l.get("exchange", {}) if isinstance(_l.get("exchange", {}), dict) else {}

    # API Keys: Prioritize ENV, fall back to Config
    BINANCE_API_KEY = (
        os.getenv("BINANCE_API_KEY")
        or os.getenv("EXCHANGE_API_KEY")
        or _l.get("api_key", "")
    )
    BINANCE_SECRET_KEY = (
        os.getenv("BINANCE_SECRET_KEY")
        or os.getenv("EXCHANGE_SECRET_KEY")
        or _l.get("secret_key", "")
    )

    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    # Backward-compatible mode mapping
    _legacy_testnet = _l.get("testnet", None)
    MODE = str(_l.get("mode", "") or "").strip().lower()
    if not MODE:
        MODE = "paper" if _as_bool(_legacy_testnet, True) else "real"
    if _legacy_testnet is not None and "mode" not in _l:
        warnings.warn(
            "live.testnet is deprecated; use live.mode: paper|real. "
            "Automatically mapped for backward compatibility.",
            DeprecationWarning,
            stacklevel=2,
        )

    IS_TESTNET = MODE != "real"
    REQUIRE_REAL_ENABLE_FLAG = _as_bool(
        _l.get("require_real_enable_flag", True), True
    )

    POLL_INTERVAL = _as_int(_l.get("poll_interval", 2), 2)
    ORDER_TIMEOUT = _as_int(_l.get("order_timeout", 10), 10)
    HEARTBEAT_INTERVAL_SEC = _as_int(_l.get("heartbeat_interval_sec", 30), 30)

    EXCHANGE = {
        "driver": str(_exchange.get("driver", "ccxt")).lower(),
        "name": str(_exchange.get("name", "binance")).lower(),
        "market_type": str(_exchange.get("market_type", "future")).lower(),
        "position_mode": str(_exchange.get("position_mode", "hedge")).upper(),
        "margin_mode": str(_exchange.get("margin_mode", "isolated")).lower(),
        "leverage": _as_int(_exchange.get("leverage", 3), 3),
    }
    # Compatibility fields used by legacy code/tests
    EXCHANGE_ID = EXCHANGE["name"]
    MARKET_TYPE = EXCHANGE["market_type"]
    POSITION_MODE = EXCHANGE["position_mode"]
    MARGIN_MODE = EXCHANGE["margin_mode"]
    LEVERAGE = EXCHANGE["leverage"]

    # Optional exchange constraints (fallbacks if market metadata unavailable)
    _sym_limits = _l.get("symbol_limits", {}) if isinstance(_l.get("symbol_limits", {}), dict) else {}
    SYMBOL_LIMITS = _sym_limits

    @classmethod
    def validate(cls):
        if not cls.SYMBOLS:
            raise ValueError("No symbols configured in trading.symbols.")

        symbol_re = re.compile(r"^[A-Z0-9]+/[A-Z0-9]+$")
        for symbol in cls.SYMBOLS:
            if not symbol_re.match(symbol):
                raise ValueError(
                    f"Invalid symbol format '{symbol}'. Expected format like BTC/USDT."
                )

        if cls.MODE not in {"paper", "real"}:
            raise ValueError("live.mode must be one of: paper, real.")

        if cls.EXCHANGE["driver"] not in {"ccxt", "mt5"}:
            raise ValueError("live.exchange.driver must be 'ccxt' or 'mt5'.")

        if cls.EXCHANGE["market_type"] not in {"spot", "future"}:
            raise ValueError("live.exchange.market_type must be 'spot' or 'future'.")

        if cls.EXCHANGE["position_mode"] not in {"ONEWAY", "HEDGE"}:
            raise ValueError("live.exchange.position_mode must be ONEWAY or HEDGE.")

        if cls.EXCHANGE["margin_mode"] not in {"isolated", "cross"}:
            raise ValueError("live.exchange.margin_mode must be isolated or cross.")

        if cls.EXCHANGE["leverage"] < 1 or cls.EXCHANGE["leverage"] > 3:
            raise ValueError(
                "live.exchange.leverage must be in range [1, 3] for staged deployment."
            )

        if not cls.BINANCE_API_KEY or not cls.BINANCE_SECRET_KEY:
            raise ValueError(
                "API keys are missing. Set BINANCE_API_KEY and BINANCE_SECRET_KEY via .env/environment."
            )

        if cls.MODE == "real" and cls.REQUIRE_REAL_ENABLE_FLAG:
            real_flag = os.getenv("LUMINA_ENABLE_LIVE_REAL", "")
            if not _as_bool(real_flag, False):
                raise ValueError(
                    "Real trading is blocked by default. Set LUMINA_ENABLE_LIVE_REAL=true to allow live real mode."
                )


class OptimizationConfig:
    """
    Configuration for Optimization.
    """

    _o = _CONFIG_DATA.get("optimization", {})

    METHOD = _o.get("method", "OPTUNA")
    STRATEGY_NAME = _o.get("strategy", "RsiStrategy")

    OPTUNA_CONFIG = _o.get("optuna", {})
    GRID_CONFIG = _o.get("grid", {})
    WALK_FORWARD_FOLDS = _as_int(_o.get("walk_forward_folds", 3), 3)
    OVERFIT_PENALTY = _as_float(_o.get("overfit_penalty", 0.5), 0.5)
