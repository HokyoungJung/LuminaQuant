"""Live trading package namespace.

This package intentionally avoids eager imports so optional dependencies
(`websockets`, exchange SDKs) do not break unrelated commands.
"""

__all__ = [
    "BinanceWebSocketDataHandler",
    "LiveDataHandler",
    "LiveExecutionHandler",
    "LiveTrader",
]
