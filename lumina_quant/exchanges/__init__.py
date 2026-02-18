from .ccxt_exchange import CCXTExchange
from .mt5_exchange import MT5Exchange


def get_exchange(config: object):
    """Factory function to get the appropriate exchange implementation."""
    exchange_config = getattr(config, "EXCHANGE", None)
    if not isinstance(exchange_config, dict):
        raise ValueError("EXCHANGE config must be a dictionary with a 'driver' field.")
    driver = str(exchange_config.get("driver", "")).strip().lower()

    if driver == "ccxt":
        return CCXTExchange(config)
    if driver == "mt5":
        return MT5Exchange(config)
    raise ValueError(f"Unknown exchange driver: {driver}")
