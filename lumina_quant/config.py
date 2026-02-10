import os
import yaml
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
    INITIAL_CAPITAL = float(_t.get("initial_capital", 10000.0))
    TARGET_ALLOCATION = float(_t.get("target_allocation", 0.1))
    MIN_TRADE_QTY = float(_t.get("min_trade_qty", 0.001))


class BacktestConfig(BaseConfig):
    """
    Configuration for Backtesting.
    """

    _b = _CONFIG_DATA.get("backtest", {})

    START_DATE = _b.get("start_date", "2024-01-01")
    # Handle None for end_date safely
    END_DATE = _b.get("end_date")

    COMMISSION_RATE = float(_b.get("commission_rate", 0.001))
    SLIPPAGE_RATE = float(_b.get("slippage_rate", 0.0005))
    ANNUAL_PERIODS = int(_b.get("annual_periods", 252))
    RISK_FREE_RATE = 0.0  # Optional, default to 0


class LiveConfig(BaseConfig):
    """
    Configuration for Live Trading.
    """

    _l = _CONFIG_DATA.get("live", {})

    # API Keys: Prioritize ENV, fall back to Config
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY") or _l.get("api_key", "")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY") or _l.get("secret_key", "")

    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    IS_TESTNET = _l.get("testnet", True)
    POLL_INTERVAL = int(_l.get("poll_interval", 2))
    ORDER_TIMEOUT = int(_l.get("order_timeout", 10))

    @classmethod
    def validate(cls):
        if not cls.BINANCE_API_KEY or not cls.BINANCE_SECRET_KEY:
            raise ValueError(
                "API Keys are missing. Please set BINANCE_API_KEY and BINANCE_SECRET_KEY in .env or config.yaml."
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
