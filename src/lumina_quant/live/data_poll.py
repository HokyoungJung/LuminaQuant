"""Live poll data handler with feature-flagged data-source routing."""

from __future__ import annotations

from lumina_quant.live.data_binance_live import BinanceLiveDataHandler
from lumina_quant.live.data_materialized import CommittedWindowDataHandler


def _resolve_market_data_source(config) -> str:
    token = str(getattr(config, "MARKET_DATA_SOURCE", "committed") or "committed")
    token = token.strip().lower().replace("-", "_")
    return "binance_live" if token in {"binance_live", "binance", "live"} else "committed"


class LiveDataHandler:
    """Factory wrapper preserving legacy constructor signature."""

    def __new__(cls, events, symbol_list, config, exchange=None):
        source = _resolve_market_data_source(config)
        if source == "binance_live":
            return BinanceLiveDataHandler(
                events,
                symbol_list,
                config,
                exchange,
                transport="poll",
            )
        return CommittedWindowDataHandler(events, symbol_list, config, exchange)


__all__ = ["LiveDataHandler"]
