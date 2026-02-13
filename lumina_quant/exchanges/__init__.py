from .ccxt_exchange import CCXTExchange
from .mt5_exchange import MT5Exchange


def get_exchange(config: object):
    """Factory function to get the appropriate exchange implementation."""
    # Parse config structure
    # Supports old style (exchange_id) for backward compat or new style (exchange.driver)

    # Check for new structure first
    exchange_config = getattr(config, "EXCHANGE", None)
    if exchange_config and isinstance(exchange_config, dict):
        driver = exchange_config.get("driver", "ccxt").lower()
    else:
        # Fallback or simplified config access if config object flattens yaml
        # Assuming config object might have EXCHANGE_DRIVER or similar if flattened
        # Or check generic exchange_id
        driver = getattr(config, "EXCHANGE_ID", "binance").lower()

    # Normalize driver names
    if driver == "binance":
        driver = "ccxt"  # backward compat

    if driver == "ccxt":
        return CCXTExchange(config)
    elif driver == "mt5":
        return MT5Exchange(config)
    else:
        raise ValueError(f"Unknown exchange driver: {driver}")
