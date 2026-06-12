"""Runtime settings helpers for strategy-factory research runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lumina_quant.symbols import canonicalize_symbol_list

_DEFAULT_SYMBOL_FALLBACK: tuple[str, ...] = (
    "BTC/USDT",
    "ETH/USDT",
    "XRP/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "TRX/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "TON/USDT",
    "AVAX/USDT",
    "XAU/USDT",
    "XAG/USDT",
    "XPT/USDT",
    "XPD/USDT",
)
_DEFAULT_PARQUET_ROOT = "data/market_parquet"
_DEFAULT_EXCHANGE = "binance"


def _safe_base_config_value(name: str, default: Any) -> Any:
    try:
        from lumina_quant.configuration import get_default_runtime_config

        rt = get_default_runtime_config()
    except (AttributeError, ImportError, ModuleNotFoundError, FileNotFoundError, RuntimeError):
        return default
    # Map known uppercase names to typed RuntimeConfig fields.
    _MAP = {
        "SYMBOLS": lambda r: list(r.trading.symbols),
        "MARKET_DATA_PARQUET_PATH": lambda r: r.storage.market_data_parquet_path,
        "MARKET_DATA_EXCHANGE": lambda r: r.storage.market_data_exchange,
    }
    if name in _MAP:
        try:
            return _MAP[name](rt)
        except Exception:
            return default
    return default


def _default_market_data_settings() -> dict[str, Any]:
    return {
        "symbols": list(default_research_symbol_universe()),
        "market_data_parquet_path": str(
            _safe_base_config_value("MARKET_DATA_PARQUET_PATH", _DEFAULT_PARQUET_ROOT)
            or _DEFAULT_PARQUET_ROOT
        ),
        "market_data_exchange": str(
            _safe_base_config_value("MARKET_DATA_EXCHANGE", _DEFAULT_EXCHANGE) or _DEFAULT_EXCHANGE
        ),
    }


def default_research_symbol_universe() -> tuple[str, ...]:
    """Resolve the default strategy-factory symbol universe lazily."""
    raw_symbols = _safe_base_config_value("SYMBOLS", _DEFAULT_SYMBOL_FALLBACK)
    try:
        return tuple(canonicalize_symbol_list(list(raw_symbols)))
    except (TypeError, ValueError):
        return _DEFAULT_SYMBOL_FALLBACK


def current_research_market_data_settings(
    runtime_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the explicit market-data settings used by research helpers."""
    if runtime_settings is not None:
        defaults = dict(runtime_settings)
    else:
        try:
            from lumina_quant.configuration import current_market_data_runtime_settings
        except (AttributeError, ImportError, ModuleNotFoundError):
            defaults = _default_market_data_settings()
        else:
            try:
                defaults = current_market_data_runtime_settings()
            except FileNotFoundError:
                defaults = _default_market_data_settings()

    return {
        "symbols": canonicalize_symbol_list(list(defaults["symbols"])),
        "parquet_root": str(
            defaults.get(
                "market_data_parquet_path", defaults.get("parquet_root", _DEFAULT_PARQUET_ROOT)
            )
            or _DEFAULT_PARQUET_ROOT
        ),
        "exchange": str(
            defaults.get("market_data_exchange", defaults.get("exchange", _DEFAULT_EXCHANGE))
            or _DEFAULT_EXCHANGE
        ),
    }
